" Set global variables:
let g:netrw_banner = 0
let g:netrw_liststyle = 3
let g:netrw_browse_split = 4
let g:netrw_altv = 1
let g:netrw_winsize = 25

" Configure workspace:
colorscheme desert
set nowrap
set guioptions+=b
set number

" Configure initial layout:
augroup ProjectDrawer
	autocmd!
	autocmd VimEnter * :Vexplore
	autocmd VimEnter * wincmd w
augroup END
set foldmethod=syntax
set foldcolumn=3
set nofoldenable

" Shortcut to open terminal window below all splits:
cabbrev bterm botright terminal ++rows=15
